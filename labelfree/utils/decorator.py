import numpy as np
import pandas as pd
import polars as pl
from functools import wraps
import inspect
from typing import Callable, Any, TypeVar, Union

# For more precise type hints
PandasDataFrame = TypeVar("pd.DataFrame")
PolarsDataFrame = TypeVar("pl.DataFrame")
NumpyArray = TypeVar("np.ndarray")
InputDataType = Union[PandasDataFrame, PolarsDataFrame, NumpyArray]
OutputDataType = Any  # The wrapped function can return various types


def as_numpy_array(param_name: str) -> Callable:
    """
    A decorator that converts a specified DataFrame/array argument of a function
    to a NumPy array. It then attempts to convert the NumPy array result back
    to the original DataFrame type (pandas or polars), preserving column names
    and index (for pandas) if dimensions are compatible.

    Args:
        param_name (str): The name of the function parameter to convert.

    Returns:
        Callable: The wrapper function.
    """

    def decorator(func: Callable[..., OutputDataType]) -> Callable[..., OutputDataType]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> OutputDataType:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if param_name not in bound_args.arguments:
                raise ValueError(
                    f"Parameter '{param_name}' not found in function signature: {list(bound_args.arguments.keys())}."
                )

            input_arg = bound_args.arguments[param_name]
            original_type = type(input_arg)
            original_columns = None
            original_index = None  # For pandas
            original_shape = None

            # Convert to NumPy array
            if isinstance(input_arg, pd.DataFrame):
                original_columns = input_arg.columns
                original_index = input_arg.index
                original_shape = input_arg.shape
                numpy_array = input_arg.to_numpy()
            elif isinstance(input_arg, pl.DataFrame):
                original_columns = input_arg.columns
                original_shape = input_arg.shape
                # Polars .to_numpy() might not be zero-copy in all cases, but it's the standard way
                numpy_array = input_arg.to_numpy()
            elif isinstance(input_arg, np.ndarray):
                original_shape = input_arg.shape
                numpy_array = input_arg
                # If it's a 1D numpy array that might have come from a single-column DataFrame,
                # we might not have original_columns. This is a limitation.
            else:
                # If not a supported DataFrame type or numpy array, pass as is.
                # The wrapped function will have to handle it.
                # Or, one could raise a TypeError here.
                # For this rewrite, we'll pass it through, similar to original behavior.
                print(
                    f"Warning: Input type {original_type} for parameter '{param_name}' "
                    "is not a pandas DataFrame, polars DataFrame, or NumPy array. "
                    "Passing as is."
                )
                numpy_array = input_arg  # Pass as is

            # Replace the parameter with the numpy array
            bound_args.arguments[param_name] = numpy_array

            # Call the decorated function
            result = func(*bound_args.args, **bound_args.kwargs)

            # Attempt to reconstruct output if the result is a NumPy array
            # and the original input was a DataFrame
            if not isinstance(result, np.ndarray) or original_shape is None:
                return result  # Return as is if not a NumPy array or no original shape info

            # --- Pandas DataFrame Reconstruction ---
            if original_type == pd.DataFrame:
                # Case 1: Result is 1D array, original had 1 column or was 1D-like.
                # Convert to Series if length matches original number of rows.
                if result.ndim == 1 and result.shape[0] == original_shape[0]:
                    series_name = (
                        original_columns[0] if len(original_columns) == 1 else None
                    )
                    # If original_index is None (e.g. from np.array input that looked like a Series),
                    # then it will default to RangeIndex.
                    return pd.Series(result, index=original_index, name=series_name)

                # Case 2: Result is 2D array.
                elif result.ndim == 2:
                    # Reconstruct DataFrame if rows match and columns match
                    if result.shape[0] == original_shape[0] and result.shape[1] == len(
                        original_columns
                    ):
                        return pd.DataFrame(
                            result, index=original_index, columns=original_columns
                        )
                    # Reconstruct DataFrame if rows match and original was single column (result can be multi-column)
                    # This allows functions that expand features from a single column
                    elif (
                        result.shape[0] == original_shape[0]
                        and len(original_columns) == 1
                    ):
                        # Heuristic: if result has multiple columns, we can't use the single original_column name.
                        # Best to return a DataFrame with default column names or let user handle.
                        # For simplicity, we'll return a DF with default columns if result.shape[1] > 1
                        # If result.shape[1] == 1, we can use original_columns[0]
                        cols_to_use = original_columns if result.shape[1] == 1 else None
                        return pd.DataFrame(
                            result, index=original_index, columns=cols_to_use
                        )

                # Fallback: return NumPy array if no clear reconstruction path
                return result

            # --- Polars DataFrame Reconstruction ---
            elif original_type == pl.DataFrame:
                # Case 1: Result is 1D array, attempt to make a single-column DataFrame
                if result.ndim == 1 and result.shape[0] == original_shape[0]:
                    col_name = (
                        original_columns[0]
                        if original_columns and len(original_columns) == 1
                        else "output"
                    )
                    try:
                        return pl.DataFrame({col_name: result})
                    except Exception as e:  # Polars can be strict
                        print(
                            f"Warning: Could not create polars DataFrame from 1D result: {e}"
                        )
                        return result  # Fallback

                # Case 2: Result is 2D array
                elif result.ndim == 2:
                    # Reconstruct if rows and number of columns match
                    if result.shape[0] == original_shape[0] and result.shape[1] == len(
                        original_columns
                    ):
                        try:
                            return pl.DataFrame(result, schema=original_columns)
                        except Exception as e:
                            print(
                                f"Warning: Could not create polars DataFrame with original schema: {e}"
                            )
                            # Try with generic column names if schema fails but shape matches
                            return pl.DataFrame(
                                result,
                                schema=[f"col_{i}" for i in range(result.shape[1])],
                            )
                    # Reconstruct if rows match, original was single column (result can be multi-column)
                    elif (
                        result.shape[0] == original_shape[0]
                        and len(original_columns) == 1
                        and result.shape[1] > 0
                    ):
                        # Try to create a DataFrame with new column names
                        try:
                            return pl.DataFrame(
                                result,
                                schema=[
                                    (
                                        f"{original_columns[0]}_{i}"
                                        if i == 0
                                        else f"col_{i}"
                                    )
                                    for i in range(result.shape[1])
                                ],
                            )
                        except Exception as e:
                            print(
                                f"Warning: Could not create polars DataFrame from 2D result (single col input): {e}"
                            )
                            return result

                # Fallback: return NumPy array if no clear reconstruction path
                return result

            # If original_type was np.ndarray, or other, result is already np.ndarray (or as is)
            return result

        return wrapper

    return decorator
