"""
@File - pipeline.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

from .config import load_config
from .run_context import init_run
from .ingest import ingest_incoming


def main():
    config = load_config()
    run_context = init_run(config)
    df = ingest_incoming(
        run_context.run_id, config.paths.incoming, config.excel.extensions
    )


if __name__ == "__main__":
    main()
