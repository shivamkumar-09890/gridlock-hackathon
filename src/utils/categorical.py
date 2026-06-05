def convert_cat_columns(
    df,
    cat_cols
):
    df = df.copy()

    for col in cat_cols:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype("category")
            )

    return df