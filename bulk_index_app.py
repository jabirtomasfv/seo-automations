import streamlit as st
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

def index_api(request_id, response, exception):
    if exception is not None:
        st.error(f"Error: {exception}")
    else:
        st.success(f"URL processed: {response['urlNotificationMetadata']['url']}")

def get_notification_status(service, urls):
    statuses = []
    for url in urls:
        try:
            response = service.urlNotifications().getMetadata(url=url).execute()
            statuses.append((url, response['latestUpdate']['type'], response['latestUpdate']['notifyTime']))
        except Exception as e:
            statuses.append((url, "Error", str(e)))
    return statuses

def main():
    st.title("Google Indexing API - Bulk URL Submission")

    json_key = st.file_uploader("Upload your service account JSON key file", type="json")

    if json_key:
        try:
            json_key_content = json.load(json_key)
            st.success("JSON key file loaded successfully")
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON: {str(e)}")
            st.write("File content:")
            st.write(json_key.getvalue().decode("utf-8"))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    tab1, tab2 = st.tabs(["Submit URLs", "Check Status"])

    with tab1:
        url_file = st.file_uploader("Upload a file with URLs (one per line)", type="txt")
        request_type = st.radio("Select request type:", ("URL_UPDATED", "URL_DELETED"))

        if json_key and url_file:
            json_key_content = json.load(json_key)
            urls = [line.decode('utf-8').strip() for line in url_file]
            requests = {url: request_type for url in urls}

            credentials = service_account.Credentials.from_service_account_info(
                json_key_content, scopes=["https://www.googleapis.com/auth/indexing"])
            service = build('indexing', 'v3', credentials=credentials)

            batch = service.new_batch_http_request(callback=index_api)

            for url, api_type in requests.items():
                batch.add(service.urlNotifications().publish(
                    body={"url": url, "type": api_type}))

            if st.button("Submit URLs"):
                with st.spinner(f"Submitting URLs for {'updating' if request_type == 'URL_UPDATED' else 'removal'}..."):
                    batch.execute()
                st.success("Batch request completed!")

    with tab2:
        status_file = st.file_uploader("Upload a file with URLs to check status (one per line)", type="txt")
        
        if json_key and status_file:
            json_key_content = json.load(json_key)
            urls_to_check = [line.decode('utf-8').strip() for line in status_file]

            credentials = service_account.Credentials.from_service_account_info(
                json_key_content, scopes=["https://www.googleapis.com/auth/indexing"])
            service = build('indexing', 'v3', credentials=credentials)

            if st.button("Check Status"):
                with st.spinner("Checking URL statuses..."):
                    statuses = get_notification_status(service, urls_to_check)
                
                st.subheader("Notification Statuses")
                for url, status, time in statuses:
                    st.write(f"URL: {url}")
                    st.write(f"Status: {status}")
                    st.write(f"Time: {time}")
                    st.write("---")

if __name__ == "__main__":
    main()
