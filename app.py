import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

# Set page configuration (Streamlit's up-to-date syntax)
st.set_page_config(page_title="24x24 g Matrix Calculator", layout="wide")

st.title("24x24 g Matrix Calculator")
st.markdown("""
This app computes the **g** matrix defined as:
\[
g(l,m,n,p,q,r) = \cos\!\Bigl(\frac{2\pi}{24}(pl+qm+rn \bmod 24)\Bigr) - i\,\sin\!\Bigl(\frac{2\pi}{24}(pl+qm+rn \bmod 24)\Bigr)
\]
using the input data provided in a spreadsheet.  
The app then computes:
- The complex conjugate matrix \(G_{\text{cc}}\),
- The product \(G \times G_{\text{cc}}\), and
- The transpose of \(G_{\text{cc}}\).
""")

st.markdown("### Upload Your Spreadsheet")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

@st.cache_data
def load_data(file: BytesIO, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

if uploaded_file is not None:
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
    st.write("File Details:", file_details)
    
    # Read file based on type
    if uploaded_file.name.endswith('.csv'):
        df = load_data(uploaded_file, "csv")
    else:
        df = load_data(uploaded_file, "xlsx")
    
    st.markdown("### Input Data Preview")
    st.dataframe(df)
    
    # Check that the required columns exist.
    required_cols = ["index", "l", "m", "n", "p", "q", "r"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Uploaded file must contain the following columns: {required_cols}")
    else:
        # Ensure exactly 24 rows are present.
        if df.shape[0] != 24:
            st.warning("Warning: The data does not contain exactly 24 rows. The calculations assume 24 rows (indexed 1 to 24).")
        
        # For our computation we assume:
        # - The row index i (0-based here) uses (l, m, n) from that row.
        # - The column index j uses (p, q, r) from the jth row.
        l_arr = df["l"].to_numpy()
        m_arr = df["m"].to_numpy()
        n_arr = df["n"].to_numpy()
        p_arr = df["p"].to_numpy()
        q_arr = df["q"].to_numpy()
        r_arr = df["r"].to_numpy()
        
        # Define function to compute each matrix element.
        def compute_entry(l_val, m_val, n_val, p_val, q_val, r_val):
            # Compute inner product and reduce modulo 24:
            prod = (p_val * l_val + q_val * m_val + r_val * n_val) % 24
            angle = (2 * np.pi / 24) * prod
            return np.cos(angle) - 1j * np.sin(angle)
        
        # Initialize matrices
        G = np.empty((24, 24), dtype=complex)
        for i in range(24):
            for j in range(24):
                G[i, j] = compute_entry(l_arr[i], m_arr[i], n_arr[i],
                                        p_arr[j], q_arr[j], r_arr[j])
        
        # Compute complex conjugate of G:
        G_cc = np.conjugate(G)
        
        # Multiply G by G_cc (standard matrix multiplication)
        G_product = G @ G_cc
        
        # Compute the transpose of G_cc:
        G_cc_T = G_cc.T
        
        st.markdown("### Results")
        
        with st.expander("Show G Matrix (24×24)"):
            st.dataframe(pd.DataFrame(G.astype(str)))
        
        with st.expander("Show Complex Conjugate Matrix G_cc (24×24)"):
            st.dataframe(pd.DataFrame(G_cc.astype(str)))
        
        with st.expander("Show Product G × G_cc (24×24)"):
            st.dataframe(pd.DataFrame(G_product.astype(str)))
        
        with st.expander("Show Transpose of G_cc (24×24)"):
            st.dataframe(pd.DataFrame(G_cc_T.astype(str)))
            
        st.success("Calculations complete!")
