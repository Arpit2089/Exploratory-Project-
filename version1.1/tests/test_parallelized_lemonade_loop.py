# tests/test_parallel_children.py
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

class DummyChild:
    def __init__(self, id):
        self.id = id
    def evaluate_expensive(self, *args, **kwargs):
        print(f"worker: child {self.id} training start")
        time.sleep(1 + (self.id % 3))
        print(f"worker: child {self.id} training done")
        # pretend we set some attribute:
        self.trained = True

def _worker_dummy(pickled_child):
    import pickle
    child = pickle.loads(pickled_child)
    child.evaluate_expensive()
    return pickle.dumps(child)

def main():
    children = [DummyChild(i) for i in range(10)]
    pickled_children = [pickle.dumps(c) for c in children]
    max_workers = min(os.cpu_count() or 1, 4)

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_worker_dummy, pc) for pc in pickled_children]
        for fut in as_completed(futures):
            pc = fut.result()
            ch = pickle.loads(pc)
            print("returned child", ch.id, getattr(ch, 'trained', False))

if __name__ == "__main__":
    main()