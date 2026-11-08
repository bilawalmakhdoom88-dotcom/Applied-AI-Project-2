# Brain Tumor Classification API

*Project 2, Week 8 | Bilawal Makhdoom | Introduction to Applied AI*

A small web service that looks at a brain MRI slice and a short clinical note, and answers with one JSON record: what the model sees, what the note says, and whether a human should look at the case (`REFER`) or not (`ROUTINE`).

> Educational project. Decision support only; a qualified human decides. Not for diagnosis.

## At a glance
| | |
|---|---|
| Task | MRI slice: `tumor` vs `no_tumor` |
| Model | ResNet-18, fine-tuned from ImageNet (Week 6) |
| Service | FastAPI: `/health`, `/predict`, `/analyze` |
| Tests | 8 passed |
| Speed (1 CPU worker) | p50 48 ms, p95 52 ms |
| Operating threshold | 0.986 on P(tumor) |
| Run it | `uvicorn app:app --port 8001`, then open `/docs` |

---

## 1. Why this service exists
A model in a notebook can only be used by its author. As an API, any program (a hospital system, a mobile app, a chat bot) can send an image and get a structured answer, without touching the Python code or the weights. The note analysis adds the second piece of evidence a clinician would read: symptoms such as headache or vomiting.

## 2. The data and what the audit found
| Step | Result |
|---|---|
| Source | Kaggle `bilawalmakhdoom/brain-tumor`, 7,200 slices (glioma, meningioma, pituitary, notumor; 1,800 each) |
| Framing | Three tumour folders merged into `tumor` (5,400); `notumor` is `no_tumor` (1,800); ratio 3 : 1 |
| Exact duplicates | 340 files in 153 sets, so 187 extra copies were removed |
| Near-duplicates | Fingerprint audit removed 260 validation and 230 test images that had a twin in training |
| Final split | train 4,870, validation 909, test 744 (57 no_tumor, 687 tumor) |

Most important finding: the near-duplicate audit showed that three quarters of the healthy validation images had a twin in the training set. Without the cleaning, validation scores would have been inflated.

## 3. How well the model works
Training took about two minutes on a GPU (best validation macro-F1 0.924). The test set was used once.

**Default cut-off (about 0.5)**

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| no_tumor | 0.962 | 0.895 | 0.927 | 57 |
| tumor | 0.991 | 0.997 | 0.994 | 687 |

Accuracy 0.989, macro-F1 0.961. Confusion matrix: 51 and 6 in the no_tumor row, 2 and 685 in the tumor row (8 errors in 744).

**Tuned cut-off 0.986** (picked on validation for tumour recall of at least 0.95)

| Caught tumours | Missed tumours | Correct healthy | False alarms |
|---|---|---|---|
| 665 | 22 | 55 | 2 |

Sensitivity 0.968, specificity 0.965, PPV 0.997, NPV 0.714, AUC 0.994 (tumour prevalence in the test set: 0.923).

**What the two cut-offs mean.** Moving to 0.986 cuts false alarms from 6 to 2, but misses go from 2 to 22. Since missing a tumour is the costly error, a lower threshold or a recall target of 0.99 would be safer. The service uses 0.986 to match the reported results and exposes it as `THRESHOLD` so it can be changed without editing code.

## 4. How the service is built
- **Load once.** The checkpoint (about 44 MB) is read at startup, not per request.
- **Same preprocessing as training.** Grayscale, 224 x 224 and ImageNet normalisation come from the settings saved inside the checkpoint.
- **Keep the server free.** Model inference runs in a thread pool, so a slow prediction does not block `/health` or other requests.
- **Assume bad input.** Only JPEG/PNG, at most 5 MB, decoded safely, with a clear 4xx message for every failure.
- **Mask first.** CNIC and mobile numbers in the note are replaced with `[CNIC]` and `[PHONE]` before anything else reads the note.
- **Decision rule.** `REFER` if P(tumor) reaches the threshold or the note contains an urgent word (headache, seizure, vomiting, nausea, dizziness, vision, weakness, numbness, confusion, memory, speech, balance); otherwise `ROUTINE`.

## 5. Run it yourself
```
python -m venv .venv
.venv\Scripts\activate                  (Mac/Linux: source .venv/bin/activate)
python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
uvicorn app:app --port 8001
```
Put `week6_model.pt` (a single file, not extracted) and `sample.jpeg` in the same folder. Then open http://127.0.0.1:8001/docs and press **Try it out**.

Settings (environment variables): `MODEL_PATH` (default `week6_model.pt`), `POSITIVE` (default `tumor`), `THRESHOLD` (default `0.986`), `API_KEY` (optional; if set, send it in the `X-API-Key` header).

## 6. Example call
```
curl -X POST http://127.0.0.1:8001/analyze ^
     -F "file=@sample.jpeg;type=image/jpeg" ^
     -F "note=Headache and vomiting since Monday. Call 0300-1234567"
```
(On Mac/Linux, end lines with `\` instead of `^`.)

Actual response:
```json
{
  "prediction": {
    "label": "tumor",
    "probabilities": { "no_tumor": 0.004, "tumor": 0.996 },
    "latency_ms": 52.3
  },
  "note_masked": "Headache and vomiting since Monday. Call [PHONE]",
  "fields": {},
  "urgent_terms": ["headache", "vomiting"],
  "decision": "REFER",
  "reasons": ["image: P(tumor) = 1.00", "note mentions: headache, vomiting"],
  "disclaimer": "Decision support only; a qualified human decides."
}
```
`/predict` returns just the `prediction` part. `fields` is empty because the phone number is masked before extraction.

## 7. What happens with bad input
| Input | Status | Message |
|---|---|---|
| PDF or text file | 415 | Upload a JPEG or PNG image |
| Image over 5 MB | 413 | Image larger than 5 MB |
| Unreadable image | 400 | File is not a readable image |
| Missing file or note | 422 | FastAPI validation error |
| Note over 2,000 characters | 422 | Note longer than 2,000 characters |
| Wrong API key (if enabled) | 401 | Invalid or missing API key |

Tried by hand: uploading `app.py` as `text/plain` to `/predict` returned `{"detail":"Upload a JPEG or PNG image"}`.

## 8. Tests
```
pytest -q test_api.py
8 passed, 1 warning in 21.02s
```
(The warning is a library deprecation notice and does not affect results.)

The eight tests cover: the server starts with the model loaded; `/predict` returns a valid answer whose probabilities sum to 1; the API gives the same probabilities as calling the model directly; wrong type (415), oversize (413), corrupt file (400) and missing file (422) are rejected cleanly; and the phone number and CNIC never appear anywhere in the `/analyze` response.

## 9. Speed and capacity
Setup: 50 requests one after another, one worker, CPU only. Laptop: AMD Ryzen 5 7430U (6 cores), 7.4 GB RAM, Windows. (Borrowed laptop; mine was unavailable.)

```
n=50 p50=48 ms p95=52 ms max=98 ms
Throughput with 1 worker ~ 20.5 requests/s
```

Little's law: one worker handles about 1000 / 48 ≈ 20.8 requests per second, which is roughly 1.8 million per day if requests arrived evenly. Real traffic comes in bursts and from many clients at once, so the safe figure is lower, and the tail (p95) would be longer than this single-client test shows.

## 10. Check Your Understanding
**1. A client uploads a PDF to /predict. Which status code, and why not 500?**
415 (Unsupported Media Type). The client sent the wrong kind of file and can fix it, so it is a 4xx error. A 500 would say the server itself broke, which is untrue and gives the client nothing to act on.

**2. Why load the model in lifespan and not inside the endpoint?**
Reading about 44 MB of weights and rebuilding the network takes seconds, while one prediction takes tens of milliseconds. Loading once at startup means each request only pays for the prediction.

**3. p50 = 40 ms, p95 = 400 ms. What does it tell you?**
Most requests are quick, but about one in twenty is ten times slower, and an average would hide that. I would look for requests queuing behind each other, a blocked event loop, cold starts, other programs using the CPU, and unusually large uploads.

**4. 500,000 requests a day, peak 5 times the average, 25 requests/s per worker?**
Average: 500,000 / 86,400 ≈ 5.8 requests/s. Peak: 5 x 5.8 ≈ 29 requests/s. Workers: 29 / 25 ≈ 1.16, so 2 workers.

**5. Why can inference in an async def endpoint slow other requests?**
An async server runs on one event loop that only switches tasks at an `await`. Model inference is heavy CPU work with no `await`, so the loop is stuck until it finishes and everything else, even `/health`, waits. Moving it to a thread pool frees the loop.

**6. Communication channels in a 4-person team?**
n(n - 1) / 2 = 4 x 3 / 2 = 6.

## 11. What I learned while building it
- **A virtual environment remembers its folder.** After I renamed the project folder, `pip` stopped working with a launcher error because `.venv` still pointed to the old path. Deleting `.venv` and recreating it fixed it, and now I decide the folder name before creating the environment.
- **Open the terminal inside the project folder.** My first attempts ran from `C:\` and from inside an unextracted zip, which created the environment in the wrong place or could not find the folder.
- **One broken quote stops the whole server.** When I changed the API title, a leftover fragment of the old text caused a `SyntaxError` and uvicorn would not start; replacing the whole statement fixed it.
- **Measure before trusting.** The audit found 340 exact duplicates and many near-duplicates, and the benchmark showed a very small gap between p50 and p95 for sequential requests.
- **A better operating point on paper is not always safer.** The tuned threshold reduces false alarms but misses 22 tumours instead of 2, so I now report both operating points together.

## 12. Limits
- No patient IDs in the file names, so slices from one patient may be in different splits and the scores may be optimistic.
- Only 57 healthy test images: one image moves that recall by about 1.8 points.
- One split and one random seed; no cross-validation or confidence intervals.
- Binary framing hides whether glioma, meningioma or pituitary tumours are detected worse than each other.
- Public data only; no clinical validation.
- At 0.986 the service misses 22 of 687 test tumours. Also, `label` follows the 0.5 rule while `decision` follows the threshold, so a slice with P(tumor) between 0.5 and 0.986 can be labelled `tumor` and still get `ROUTINE` unless the note adds a reason.
- Not a medical device; never a replacement for a radiologist.
