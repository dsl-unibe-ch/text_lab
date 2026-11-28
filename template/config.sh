# Load a Text Lab session's configuration: the developer overrides in
# dev.env (sandbox apps only), then the site configuration.
#
# Open OnDemand copies template/ into the session's job folder. Two scripts
# there source this file: script.sh.erb, which starts the app, and
# before.sh.erb, which only needs to know the interface the session runs
# (TEXT_LAB_UI) for the Connect button.
#
# Nothing here is specific to a cluster except the default location of the
# live site configuration (TEXT_LAB_SITE_ENV below), the only site-specific
# value in the files the HPC team deploys.

config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional developer overrides for a sandbox app, e.g. TEXT_LAB_SRC pointing
# at a working tree or TEXT_LAB_SITE_ENV at the repository's site.env. It is
# gitignored and never part of a production deployment; see
# template/dev.env.example.
if [[ -f "$config_dir/dev.env" ]]; then
  echo "Loading developer overrides from $config_dir/dev.env"
  # shellcheck source=/dev/null
  source "$config_dir/dev.env"
fi

# The site configuration (deploy/site.env in the repository).
TEXT_LAB_SITE_ENV="${TEXT_LAB_SITE_ENV:-/storage/research/dsl_shared/solutions/ondemand/text_lab/site.env}"
if [[ ! -f "$TEXT_LAB_SITE_ENV" ]]; then
  echo "Error: site configuration not found at $TEXT_LAB_SITE_ENV" >&2
  return 1
fi
# shellcheck source=/dev/null
source "$TEXT_LAB_SITE_ENV"
