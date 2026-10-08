"""DataSphere CLI with only the repetitive IAM-token warning filtered out."""
import logging


class TokenWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != (
            "iam token from env var is not refreshable, so it may expire over time"
        )


if __name__ == "__main__":
    # Attach to the logger, so DataSphere's handler setup does not remove it.
    logging.getLogger("datasphere.auth").addFilter(TokenWarningFilter())
    from datasphere.main import main

    main()
