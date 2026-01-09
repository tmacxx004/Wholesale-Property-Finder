import streamlit as st
import pandas as pd
from io import StringIO
import tempfile
import os

from hennepin_mpls_deal_crawler import run

st.title("Hennepin / Minneapolis Deal Finder")

cities = st.multiselect(
    "Filter cities/suburbs (Hennepin MUNIC_NM values)",
    ["Minneapolis", "Bloomington", "Brooklyn Park", "Richfield", "Plymouth", "Edina", "Eden Prairie"],
)
enable_mpls = st.checkbox("Stack Minneapolis VBR + Violations (Tableau)", value=True)
top_n = st.slider("Top N leads", 100, 3000, 1000, 100)

if st.button("Run crawler"):
    with st.spinner("Running..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.close()

        run(
            cities=cities if cities else None,
            enable_mpls=enable_mpls,
            top_n=top_n,
            out_csv=tmp.name,
        )
        df = pd.read_csv(tmp.name)
        os.unlink(tmp.name)

    st.success(f"Done. Rows: {len(df)}")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, "ranked_leads.csv", "text/csv")
