for fun in count_zero_metoden loop_metoden; do
    echo -n "$fun "
    docker run --rm -it -v $(pwd):/app -w /app -e PYTHONDONTWRITEBYTECODE=1 python -m timeit -s 'import benchmark_test as s' "s.$fun()"
done
