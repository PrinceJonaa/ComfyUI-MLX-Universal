import tracemalloc
import timeit


def with_list_comp(n):
    keys_to_pop = [str(i) for i in range(n)]
    state_dict = {k: 1 for k in keys_to_pop}
    tracemalloc.start()
    [state_dict.pop(k) for k in keys_to_pop]
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def with_for_loop(n):
    keys_to_pop = [str(i) for i in range(n)]
    state_dict = {k: 1 for k in keys_to_pop}
    tracemalloc.start()
    for k in keys_to_pop:
        state_dict.pop(k)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


n_elements = 10000

print(
    f"List comprehension peak memory for {n_elements} elements: {with_list_comp(n_elements) / 1024:.2f} KB"
)
print(
    f"For loop peak memory for {n_elements} elements: {with_for_loop(n_elements) / 1024:.2f} KB"
)

setup_comp = """
keys_to_pop = [str(i) for i in range(10000)]
state_dict = {k: 1 for k in keys_to_pop}
"""
stmt_comp = """
[state_dict.pop(k) for k in keys_to_pop]
"""

setup_for = """
keys_to_pop = [str(i) for i in range(10000)]
state_dict = {k: 1 for k in keys_to_pop}
"""
stmt_for = """
for k in keys_to_pop:
    state_dict.pop(k)
"""

time_comp = timeit.timeit(stmt_comp, setup=setup_comp, number=1)
time_for = timeit.timeit(stmt_for, setup=setup_for, number=1)

print(f"List comprehension time: {time_comp:.6f} seconds")
print(f"For loop time: {time_for:.6f} seconds")
