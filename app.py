"""
Install with:
    pip install streamlit pandas pyarrow openai
"""
import streamlit as st
import pandas as pd
import dq_tool
import io
import tempfile
import llm_advisor
import openai

st.title("Data Quality Assessment Tool")

# Sidebar for OpenAI API key
st.sidebar.header("Settings")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if openai_api_key:
    st.session_state['openai_api_key'] = openai_api_key

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

llm_output = None
ai_patterns = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    # Run checks
    completeness_df = dq_tool.check_completeness(df)
    expected_types = dq_tool.infer_expected_types(df)
    type_accuracy_df = dq_tool.check_type_accuracy(df, expected_types)
    pattern_df = dq_tool.check_pattern_recognition(df)

    # Merge results
    merged = completeness_df.merge(type_accuracy_df, on='column', how='outer')
    merged = merged.merge(pattern_df, on='column', how='outer')

    def completeness_color(val):
        if pd.isna(val):
            return ''
        if val == 100:
            color = 'background-color: #b6fcb6'  # green
        elif 50 <= val < 100:
            color = 'background-color: #fffcb6'  # yellow
        else:
            color = 'background-color: #fcb6b6'  # red
        return color

    # AI Analyze Columns button
    if st.button("AI Analyze Columns"):
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
        else:
            ai_patterns = {}
            with st.spinner("Contacting LLM for column pattern analysis..."):
                client = openai.OpenAI(api_key=openai_api_key)
                for col in df.columns:
                    sample = df[col].dropna().astype(str).sample(min(10, len(df[col].dropna())), random_state=42)
                    prompt = (
                        "Given these sample values, what is the most likely semantic type or pattern? "
                        "Respond with a short label (e.g., email, phone, name, address, product code, etc.):\n"
                        + "\n".join(sample)
                    )
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=50,
                            temperature=0.2,
                        )
                        ai_patterns[col] = response.choices[0].message.content.strip()
                    except Exception as e:
                        ai_patterns[col] = f"LLM error: {e}"
            merged['ai_pattern'] = merged['column'].map(ai_patterns)

    if ai_patterns is not None:
        st.success("AI pattern recognition complete. See 'ai_pattern' column in the table.")

    styled = merged.style.applymap(completeness_color, subset=['completeness_pct'])

    st.write("Data Quality Report")
    st.dataframe(styled)

    # Download buttons
    csv = merged.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "dq_report.csv", "text/csv")

    json_str = merged.to_json(orient='records', indent=2)
    st.download_button("Download JSON", json_str, "dq_report.json", "application/json")

    html_str = merged.to_html(index=False)
    st.download_button("Download HTML", html_str, "dq_report.html", "text/html")

    # Visualizations
    st.subheader("Completeness Percentage by Column")
    if 'completeness_pct' in merged.columns:
        st.bar_chart(merged.set_index('column')['completeness_pct'])

    st.subheader("Type Accuracy Percentage by Column")
    if 'accuracy_pct' in merged.columns:
        st.bar_chart(merged.set_index('column')['accuracy_pct'])

    # LLM Advisor Button with Use Case
    st.subheader("LLM Data Quality Advisor")
    use_case = st.text_input("Describe your use case (e.g., 'customer churn prediction', 'email marketing', etc.)")
    if st.button("Suggest Fixes with LLM"):
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
        elif not use_case:
            st.warning("Please describe your use case above.")
        else:
            with st.spinner("Contacting LLM advisor..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w', encoding='utf-8') as tmp:
                    merged.to_json(tmp, orient='records', indent=2)
                    tmp_path = tmp.name
                try:
                    import sys
                    from io import StringIO
                    old_stdout = sys.stdout
                    sys.stdout = mystdout = StringIO()
                    llm_advisor.suggest_fixes(tmp_path, openai_api_key=openai_api_key, use_case=use_case)
                    sys.stdout = old_stdout
                    llm_output = mystdout.getvalue()
                    st.markdown(llm_output)
                except Exception as e:
                    st.error(f"LLM advisor failed: {e}") 