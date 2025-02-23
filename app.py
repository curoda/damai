import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

# Configure the Streamlit page
st.set_page_config(page_title="24x24 g Matrix Calculator", layout="wide")

st.title("24x24 g Matrix Calculator")
st.markdown(r"""
This app computes the **g** matrix defined as:
\[
g(l,m,n,p,q,r) = \cos\!\Bigl(\frac{2\pi}{24}\,(p\,l + q\,m + r\,n)\Bigr) - i\,\sin\!\Bigl(\frac{2\pi}{24}\,(p\,l + q\,m + r\,n)\Bigr)
\]
using the input data provided in a spreadsheet. The **g** matrix is a 24×24 matrix constructed as follows:

- For row \(i\) (from the table), use \((l_i,\, m_i,\, n_i)\).
- For column \(j\) (from the table), use \((p_j,\, q_j,\, r_j)\).
- Compute
  \[
  g_{i,j} = \cos\!\Bigl(\frac{2\pi}{24}(p_j\,l_i + q_j\,m_i + r_j\,n_i)\Bigr) - i\,\sin\!\Bigl(\frac{2\pi}{24}(p_j\,l_i + q_j\,m_i + r_j\,n_i)\Bigr).
  \]

The app then computes:
- The complex conjugate matrix **gcc** (by replacing \(i\) with \(-i\) in each entry),
- The transpose of **gcc**, and
- The **elementwise** product of **g** and the transposed **gcc**.
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
    
    # Read the file based on its extension
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
        
        # Extract column values
        l_arr = df["l"].to_numpy()
        m_arr = df["m"].to_numpy()
        n_arr = df["n"].to_numpy()
        p_arr = df["p"].to_numpy()
        q_arr = df["q"].to_numpy()
        r_arr = df["r"].to_numpy()
        
        # Define function to compute each matrix element (using raw integer sum)
        def compute_entry(l_val, m_val, n_val, p_val, q_val, r_val):
            product = p_val * l_val + q_val * m_val + r_val * n_val
            angle = (2 * np.pi / 24.0) * product
            return np.cos(angle) - 1j * np.sin(angle)
        
        # Build the 24x24 g matrix
        G = np.empty((24, 24), dtype=complex)
        for i in range(24):
            for j in range(24):
                G[i, j] = compute_entry(l_arr[i], m_arr[i], n_arr[i],
                                        p_arr[j], q_arr[j], r_arr[j])
        
        # Compute the complex conjugate matrix gcc
        G_cc = np.conjugate(G)
        
        # Compute the transpose of gcc
        G_cc_T = G_cc.T
        
        # Elementwise multiplication: g * (transpose of gcc)
        elementwise_product = G * G_cc_T
        
        st.markdown("### Results")
        
        with st.expander("Show g Matrix (24×24)"):
            st.dataframe(pd.DataFrame(G.astype(str)))
        
        with st.expander("Show Complex Conjugate Matrix gcc (24×24)"):
            st.dataframe(pd.DataFrame(G_cc.astype(str)))
        
        with st.expander("Show Transpose of gcc (24×24)"):
            st.dataframe(pd.DataFrame(G_cc_T.astype(str)))
        
        with st.expander("Show Elementwise Product (g * gccᵀ) (24×24)"):
            st.dataframe(pd.DataFrame(elementwise_product.astype(str)))
            
        st.success("Calculations complete!")

