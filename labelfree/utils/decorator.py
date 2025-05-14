import numpy as np
import pandas as pd
import polars as pl

from functools import wraps


def as_numpy_array(func):
    """
    A decorator that converts the first argument of a function, if it is a
    pandas DataFrame or a polars DataFrame, to a NumPy array. After the
    decorated function is executed, it converts the resulting NumPy array
    back to the original DataFrame type (pandas or polars) with original
    column names and index (for pandas).

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapper function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not args:
            raise ValueError(
                "The decorated function must have at least one positional argument."
            )

        input_arg = args[0]
        original_type = type(input_arg)
        original_columns = None
        original_index = None  # Specific to pandas

        # Convert to NumPy array
        if isinstance(input_arg, pd.DataFrame):
            original_columns = input_arg.columns
            original_index = input_arg.index
            numpy_array = input_arg.to_numpy()
        elif isinstance(input_arg, pl.DataFrame):
            original_columns = input_arg.columns
            numpy_array = input_arg.to_numpy()
        elif isinstance(input_arg, np.ndarray):
            numpy_array = input_arg
        else:
            # If the input is not one of the supported types, pass it as is.
            # Or, you could raise a TypeError here.
            print(
                f"Input type {original_type} is not a pandas DataFrame, polars DataFrame, or NumPy array. Passing as is."
            )
            numpy_array = input_arg

        # Modify args to pass the numpy_array as the first argument
        modified_args = (numpy_array,) + args[1:]

        # Call the decorated function
        result = func(*modified_args, **kwargs)

        # Reassemble if necessary
        if not isinstance(result, np.ndarray):
            print("Function did not return a NumPy array. Returning result as is.")
            return result

        if original_type == pd.DataFrame and original_columns is not None:
            # Try to reshape if the number of columns implies a different shape
            if (
                result.ndim == 1
                and len(original_columns) > 1
                and len(result) % len(original_columns) == 0
            ):
                result_df = pd.DataFrame(
                    result.reshape(-1, len(original_columns)),
                    columns=original_columns,
                    index=original_index,
                )
            elif (
                result.ndim == 2
                and result.shape[1] != len(original_columns)
                and len(original_columns) == 1
            ):
                result_df = pd.DataFrame(
                    result,
                    columns=original_columns,
                    index=(
                        original_index
                        if original_index is not None
                        and len(original_index) == len(result)
                        else None
                    ),
                )
            elif result.ndim == 2 and result.shape[1] == len(original_columns):
                result_df = pd.DataFrame(
                    result,
                    columns=original_columns,
                    index=(
                        original_index
                        if original_index is not None
                        and len(original_index) == len(result)
                        else None
                    ),
                )
            else:  # Fallback or if shapes don't match well for columns
                result_df = pd.DataFrame(
                    result,
                    index=(
                        original_index
                        if original_index is not None
                        and len(original_index) == len(result)
                        else None
                    ),
                )
                if len(result_df.columns) == len(original_columns):
                    result_df.columns = original_columns

            return result_df
        elif original_type == pl.DataFrame and original_columns is not None:
            # Polars from_numpy is more flexible, but schema matching is good
            if (
                result.ndim == 1
                and len(original_columns) > 1
                and len(result) % len(original_columns) == 0
            ):
                result_pl_df = pl.DataFrame(
                    result.reshape(-1, len(original_columns)), schema=original_columns
                )
            elif result.ndim == 2 and result.shape[1] == len(original_columns):
                result_pl_df = pl.DataFrame(result, schema=original_columns)
            else:  # Fallback if shapes don't match column names well
                try:
                    result_pl_df = pl.DataFrame(result, schema=original_columns)
                except Exception as e:
                    print(
                        f"Could not directly create polars.DataFrame with original schema due to: {e}. Creating without schema."
                    )
                    result_pl_df = pl.DataFrame(result)
                    if len(result_pl_df.columns) == len(original_columns):
                        result_pl_df.columns = (
                            original_columns  # Try to rename if column count matches
                        )

            return result_pl_df
        else:
            return result

    return wrapper
