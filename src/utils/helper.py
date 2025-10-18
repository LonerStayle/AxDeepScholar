
async def process_stream(stream_generator):
    results = []
    try:
        async for chunk in stream_generator:
            key = list(chunk.keys())[0]
            
            if key == 'agent':
                content = chunk['agent']['messages'][0].content if chunk['agent']['messages'][0].content != '' else chunk['agent']['messages'][0].additional_kwargs
                print(f"'agent': '{content}'")
            
            elif key == 'tools':
                for tool_msg in chunk['tools']['messages']:
                    print(f"'tools:': '{tool_msg.content}'")
                    
            results.append(chunk)
        return results
    except Exception as e:
        print(f'Error processing stream: {e}')
        return results
    
    
import inspect
from typing import get_type_hints, TypeVar, Type, Any

T = TypeVar("T")


def attach_auto_keys(cls: Type[T]) -> Type[T]:
    """클래스 정의 이후 자동으로 Key 클래스를 주입합니다 (TypedDict, BaseModel, MessagesState 전부 호환)."""
    annotations: dict[str, Any] = {}
    for base in reversed(cls.__mro__):

        try:
            hints = get_type_hints(base, include_extras=True)
        except Exception:
            hints = getattr(base, "__annotations__", {})
        annotations.update(hints or {})
        
    if not annotations:
        annotations = {
            k: v for k, v in vars(cls).items()
            if not k.startswith("_") and not inspect.isroutine(v)
        }

    key_cls = type(
        "Key",
        (),
        {k: k for k in annotations.keys()}
    )
    setattr(cls, "Key", key_cls)
    return cls

from datetime import datetime

def get_today_str(pattern = "%a %b %-d, %Y"):
    return datetime.now().strftime(pattern)

from pathlib import Path
def get_current_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()