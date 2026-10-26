class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, retryable: bool = False):
        super().__init__(code)  # Never put prompts, credentials or upstream bodies in exceptions.
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable


def not_found(kind: str = "资源") -> AppError:
    return AppError("NOT_FOUND", f"{kind}不存在或无权访问", 404)
