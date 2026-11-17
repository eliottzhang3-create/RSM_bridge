"""Single pdgpu-4090 job, using the same frozen 100-question manifest."""
from .submit import main

if __name__ == '__main__':
    raise SystemExit(main(world_size=1))
