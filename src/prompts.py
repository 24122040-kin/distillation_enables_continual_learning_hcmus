"""Student and teacher contexts of SDFT (paper Section 3, Algorithm 1: Ctx_S, Ctx_T).

The teacher context wording follows the template printed in Section 3 of the
paper. The student context is the plain task prompt.
"""

from typing import Optional, Tuple


def student_context(system: Optional[str], question: str) -> Tuple[Optional[str], str]:
    """Ctx_S(x): the task prompt alone.

    Parameters
    ----------
    system : str or None
        Task system prompt (Science only).
    question : str
        Task prompt x.

    Returns
    -------
    (system, user)
        Chat turns for the student.
    """
    return system, question


def teacher_context(system: Optional[str], question: str, demonstration: str) -> Tuple[Optional[str], str]:
    """Ctx_T(x, c): the task prompt followed by the expert demonstration.

    Parameters
    ----------
    system : str or None
        Task system prompt, shared with the student so that the only difference
        between the two contexts is the demonstration.
    question : str
        Task prompt x.
    demonstration : str
        Expert demonstration c.

    Returns
    -------
    (system, user)
        Chat turns for the teacher.
    """
    user = (
        f"{question}\n\n"
        "This is an example for a response to the question:\n"
        f"{demonstration}\n\n"
        "Now answer with a response of your own, including the thinking process:"
    )
    return system, user
