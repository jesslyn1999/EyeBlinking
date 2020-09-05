import ray
import time

ray.init(include_dashboard=False)

n = 2

@ray.remote
def my_function():
    print("first func: ")
    return 1


@ray.remote
def function_with_an_argument(value):
    print("second func: ")
    return value + 1


obj_ref1 = my_function.remote()
time.sleep(3)
print("HEYY OBJ REF 1")


@ray.remote
def blocking_operation():
    time.sleep(10)
    print("BLOCKING: ")


time.sleep(3)
obj_ref_block = blocking_operation.remote()
print("HEYY OBJ REF BLOCKING")

# ray.cancel(obj_ref_block)

assert ray.get(obj_ref1) == 1

# You can pass an object ref as an argument to another Ray remote function.
obj_ref2 = function_with_an_argument.remote(obj_ref1)
print("HEYY OBJ REF 2")
assert ray.get(obj_ref2) == 2
