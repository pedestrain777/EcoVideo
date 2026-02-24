from vdit.generators.base import VideoGenerator, create_generator, register_generator

# side-effect imports: ensure decorators run and generators get registered
try:
    from vdit.generators import wan_t2v as _wan_t2v  # noqa: F401
except ImportError:
    pass
try:
    from vdit.generators import ltx_t2v as _ltx_t2v  # noqa: F401
except ImportError:
    pass
try:
    from vdit.generators import cogvideo_t2v as _cogvideo_t2v  # noqa: F401
except ImportError:
    pass

__all__ = [
    "VideoGenerator",
    "create_generator",
    "register_generator",
]

