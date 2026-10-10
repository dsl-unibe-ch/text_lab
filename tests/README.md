# Cross-cutting tests

Tests of a single feature live next to it, in
`src/textlab/features/<feature>/tests/` (and `src/textlab/common/tests/` for
shared code). This folder is for tests that span the whole application, such
as checking that no user data is left on disk after a session, along with
fixtures shared by several features.

See the [testing guide](../docs/dev/testing.md) for how to run the tests.
