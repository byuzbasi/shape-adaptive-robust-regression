"""Keep the locally built Pages output independent of Jekyll processing."""
from pathlib import Path


def on_post_build(config, **kwargs):
    (Path(config["site_dir"]) / ".nojekyll").touch()
