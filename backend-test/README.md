## backend-test 
tests the backend implementation with fake eeg/lsl and using webcam

# first start with embeddings
1. cd backend-fydp6
2. rm backend-new/data/arcface_embeddings.pkl (as needed)
3. python enroll_from_video.py 

start the server: 
4. cd backend-test
5. uvicorn main:app --host 0.0.0.0 --port 8000 --reload


open http://localhost:8000/face/debug/ui in a browser
- fixation events can be marked with enter in the terminal or by clicking on button in the browser window

opening http://localhost:8000/db/face will show the json file (database)