"""
Compute RFM (Recency, Frequency, Monetary) scores per customer.
"""
import pandas as pd


def compute_rfm(orders_df: pd.DataFrame, payments_df: pd.DataFrame,
                snapshot_date=None) -> pd.DataFrame:
    """
    Build an RFM table from orders + payments.

    Parameters
    ----------
    orders_df : pd.DataFrame
        Must contain: order_id, customer_unique_id, order_purchase_timestamp,
        order_status.
    payments_df : pd.DataFrame
        Must contain: order_id, payment_value.
    snapshot_date : pd.Timestamp | None
        Reference date for recency.  Defaults to max purchase date + 1 day.

    Returns
    -------
    rfm : pd.DataFrame
        Columns: customer_unique_id, Recency, Frequency, Monetary.
    """
    # Only keep delivered orders (ignore cancelled / processing)
    df = orders_df[orders_df["order_status"] == "delivered"].copy()

    # Merge payment values
    df = df.merge(
        payments_df[["order_id", "payment_value"]],
        on="order_id",
        how="left",
    )

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    if snapshot_date is None:
        snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            Recency=("order_purchase_timestamp",
                     lambda x: (snapshot_date - x.max()).days),
            Frequency=("order_id", "nunique"),
            Monetary=("payment_value", "sum"),
        )
        .reset_index()
    )

    # Drop rows with missing monetary (shouldn't happen, but safety)
    rfm = rfm.dropna(subset=["Monetary"])
    rfm = rfm[rfm["Monetary"] > 0]

    return rfm
