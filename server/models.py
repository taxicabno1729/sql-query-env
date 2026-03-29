from pydantic import BaseModel


class SQLAction(BaseModel):
    sql_query: str


class SQLObservation(BaseModel):
    done: bool
    reward: float
    task_description: str
    schema_info: str
    feedback: str
    metadata: dict


class SQLState(BaseModel):
    task_id: str
    difficulty: str  # easy | medium | hard
    attempts: int
    max_attempts: int
    score_history: list[float]
