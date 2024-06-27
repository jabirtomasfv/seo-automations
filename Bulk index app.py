import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import BatchHttpRequest
import json

def index_api(request_id, response, exception):
    if exception is not None:
        st.error(f"Error: {exception}")
    else:
        st.success(f"Success: {response}")

def main():
    st.title("Google Indexing API - Bulk URL Submission")

    # File uploader for JSON key
    json_key = st.file_uploader("Upload your service account JSON key file", type="json")

    # File uploader for URLs
    url_file = st.file_uploader("Upload a file with URLs (one per line)", type="txt")

    if json_key and url_file:
        # Read JSON key
        json_key_content = json.load(json_key)

        # Read URLs
        urls = [line.decode('utf-8').strip() for line in url_file]

        # Create requests dictionary
        requests = {url: "URL_UPDATED" for url in urls}

        # Set up credentials and service
        credentials = service_account.Credentials.from_service_account_info(
            json_key_content, scopes=["https://www.googleapis.com/auth/indexing"])
        service = build('indexing', 'v3', credentials=credentials)

        # Create batch request
        batch = service.new_batch_http_request(callback=index_api)

        # Add URL notifications to batch
        for url, api_type in requests.items():
            batch.add(service.urlNotifications().publish(
                body={"url": url, "type": api_type}))

        # Execute batch request
        if st.button("Submit URLs for Indexing"):
            with st.spinner("Submitting URLs..."):
                batch.execute()
            st.success("Batch request completed!")

if __name__ == "__main__":
    main()