"""
Load and merge the Olist orders and payments CSVs.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

ORDERS_FILE = os.path.join(DATA_DIR, "olist_orders_dataset.csv")
PAYMENTS_FILE = os.path.join(DATA_DIR, "olist_order_payments_dataset.csv")
PAYMENTS_FILE_ALT = os.path.join(DATA_DIR, "payments.csv")
CUSTOMERS_FILE = os.path.join(DATA_DIR, "olist_customers_dataset.csv")


def _find_payments_file() -> str:
    """Return the path to the payments CSV, trying the canonical name first."""
    # Try canonical name, but verify it's actually readable (not a locked stub)
    if os.path.exists(PAYMENTS_FILE):
        try:
            size = os.path.getsize(PAYMENTS_FILE)
            if size > 100:  # legitimate file, not a 1-byte stub
                return PAYMENTS_FILE
        except OSError:
            pass
    # Fall back to alternate name
    if os.path.exists(PAYMENTS_FILE_ALT):
        return PAYMENTS_FILE_ALT
    raise FileNotFoundError(
        f"Missing payments file. Looked for:\n"
        f"  {PAYMENTS_FILE}\n  {PAYMENTS_FILE_ALT}\n"
        "Download the Olist dataset from "
        "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
        "and place the CSVs in the data/ folder."
    )


def load_and_merge():
    """
    Load orders, payments, and customers CSVs and return them.

    Returns
    -------
    orders_df : pd.DataFrame
        Orders table (with customer_unique_id merged in).
    payments_df : pd.DataFrame
        Payments table.
    """
    # Verify orders file
    if not os.path.exists(ORDERS_FILE):
        raise FileNotFoundError(
            f"Missing orders file: {ORDERS_FILE}\n"
            "Download the Olist dataset from "
            "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
            "and place the CSVs in the data/ folder."
        )

    payments_path = _find_payments_file()

    orders_df = pd.read_csv(ORDERS_FILE)
    payments_df = pd.read_csv(payments_path)

    # Merge customer_unique_id if customers file exists
    if os.path.exists(CUSTOMERS_FILE):
        customers_df = pd.read_csv(CUSTOMERS_FILE)
        orders_df = orders_df.merge(
            customers_df[["customer_id", "customer_unique_id"]],
            on="customer_id",
            how="left",
        )
    else:
        # Fall back: use customer_id as the unique identifier
        print("⚠️  Customers file not found — using customer_id instead of "
              "customer_unique_id.")
        orders_df["customer_unique_id"] = orders_df["customer_id"]

    return orders_df, payments_df
