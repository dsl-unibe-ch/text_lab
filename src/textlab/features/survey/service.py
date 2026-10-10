"""Reading filled-in questionnaires: the Survey feature's API.

Interfaces and the OCR feature call this module. There are two ways to
read responses:

- **Questionnaire batches** (:class:`QuestionnaireBatch`): a batch of the
  same paper questionnaire, filled in by different people. The blank form
  is learned from the batch itself (:mod:`.survey_template`), its answers
  are named from its printed text (:mod:`.survey_label`), and every
  questionnaire is read against it (:mod:`.survey_batch`). The OCR batch
  runs this in survey batch mode; a reviewer can then drop controls that
  are not really on the form and rebuild the exports.
- **Question-level extraction** on one recognized page
  (:func:`extract_page_forms`, :mod:`.form_extract`): experimental and
  hidden in the app; the OCR pipeline calls it with ``extract_survey``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.features.survey import survey_batch
from textlab.features.survey.form_extract import (
    SameLayoutTemplate,
    extract_page_forms,
)

__all__ = [
    "QuestionnaireBatch",
    "SameLayoutTemplate",
    "extract_page_forms",
]

#: Name of a questionnaire's own answers, next to its other outputs.
ANSWERS_FILE = "survey_answers.csv"


@dataclasses.dataclass
class QuestionnaireBatch:
    """A batch of questionnaires read against the form they share.

    Create it with :meth:`learn`, then :meth:`read` each questionnaire and
    :meth:`write_outputs` once at the end.

    Attributes:
        template: The blank form learned from the batch
            (``survey_template.SurveyTemplate``).
        readings: Each questionnaire's answers, in reading order.
        documents: Each questionnaire's recognized document, without its
            page images, by its folder in the result; used to rebuild the
            exports after a review.
    """

    template: Any
    readings: list[Any] = dataclasses.field(default_factory=list)
    documents: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def learn(
        cls,
        files: Sequence[Path],
        *,
        on_progress: ProgressCallback = no_progress,
        vl_session: Any = None,
    ) -> QuestionnaireBatch:
        """Learn the blank form from all questionnaires of a batch.

        Args:
            files: The questionnaires, as PDFs or images.
            on_progress: Receives progress updates.
            vl_session: A running PaddleOCR-VL worker
                (``ocr.vl_session.VLWorkerSession``) to read the form's
                printed text with, so a batch loads the model once.

        Returns:
            The batch, with no questionnaires read yet.
        """

        def report(fraction: float, text: str) -> None:
            on_progress(
                Progress(
                    f"Questionnaire layout — {text}",
                    min(0.99, max(0.0, fraction)),
                )
            )

        report(0.0, f"reading {len(files)} file(s)...")
        template, _blanks = survey_batch.prepare_template(
            files, label=True, progress=report, vl_session=vl_session
        )
        batch = cls(template=template)
        on_progress(
            Progress(
                f"Questionnaire layout — {batch.control_count} response "
                f"controls in {batch.answer_count} answers"
            )
        )
        return batch

    @property
    def warning(self) -> str | None:
        """A note for users when the batch is too small to learn the form."""
        return self.template.provenance.get("small_batch_warning")

    @property
    def control_count(self) -> int:
        """Number of response controls (checkboxes, circles) on the form."""
        return self.template.control_count

    @property
    def answer_count(self) -> int:
        """Number of answers: groups of controls that answer one item."""
        return len(self.template.rules)

    def read(
        self, file_path: Path, document: Any, output_dir: Path, folder: str
    ) -> set[str]:
        """Read one questionnaire's answers.

        Writes its answers to :data:`ANSWERS_FILE` in ``output_dir`` and
        adds them to ``document`` as form responses. Call
        :meth:`keep_document` once the document's own outputs are written.

        Args:
            file_path: The questionnaire.
            document: Its recognized document (``ocr.doc_ir.Document``).
            output_dir: The folder of its outputs.
            folder: That folder's path in the result, with ``/``.

        Returns:
            The ids of the document's table regions that the answers
            replace; leave them out of the document's table exports.
        """
        reading = survey_batch.read_document(file_path, self.template)
        reading.export_directory = folder
        self.readings.append(reading)
        survey_batch.safe_csv(
            survey_batch.answers_for_document(reading, self.template),
            output_dir / ANSWERS_FILE,
        )
        survey_batch.to_form_groups(reading, self.template, document)
        return survey_batch.survey_table_regions(document, self.template)

    def keep_document(self, folder: str, document: Any) -> None:
        """Keep a read questionnaire's document for rebuilding its exports.

        Its page images and searchable PDF are dropped to save memory.

        Args:
            folder: The document's folder in the result, as for
                :meth:`read`.
            document: The document; it is changed in place.
        """
        self.documents[folder] = survey_batch.slim_document(document)

    def write_outputs(self, folder: Path) -> dict[str, Any]:
        """Write the batch's tables, template and overlays to ``folder``.

        Args:
            folder: The ``survey/`` folder of the result.

        Returns:
            Counts for an overview.
        """
        return survey_batch.write_batch_outputs(
            self.readings, self.template, folder
        )

    def overlays(self) -> dict[str, bytes]:
        """Return each page of the learned form with its controls outlined.

        Returns:
            PNG data by file name, one per page.
        """
        return survey_batch.template_overlays(self.template)

    def overview(self) -> Any:
        """Return one row per answer, with how many respondents marked it.

        Returns:
            A ``pandas.DataFrame``; ``never_marked`` flags answers nobody
            marked, and ``control_ids`` lists each answer's controls,
            separated by commas.
        """
        return survey_batch.answer_overview(self.readings, self.template)

    def drop_controls(
        self, control_ids: Sequence[str], zip_bytes: bytes
    ) -> tuple[bytes, int, dict[str, Any]]:
        """Remove controls that are not on the form and rebuild the exports.

        The questionnaires are not read again: the stored readings and
        documents are exported anew.

        Args:
            control_ids: The controls to remove.
            zip_bytes: The batch's result ZIP.

        Returns:
            The rebuilt ZIP, the number of controls removed, and counts for
            an overview (``controls``, ``answer_groups``, ...).
        """
        removed = survey_batch.drop_controls(self.template, control_ids)
        rebuilt, summary = survey_batch.rebuild_exports(
            zip_bytes, self.template, self.readings, self.documents
        )
        return rebuilt, removed, summary
