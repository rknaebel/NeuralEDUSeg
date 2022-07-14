import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from neuraleduseg.config import get_argparser

import click
import spacy
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic.main import BaseModel

from neuraleduseg.model.atten_seg import AttnSegModel
from neuraleduseg.rst_edu_reader import RSTData

app = FastAPI(
    title="neuraleduseg-service",
    version="1.0.0",
    description="RST-based discourse segmentation via neural networks.",
)
templates = Jinja2Templates(directory="app/pages")

model: AttnSegModel = None
rst_data: RSTData = None
spacy_nlp: spacy.pipeline.Pipe = None


def load_models(args):
    """Load models needed for EDU segmentation."""
    rst_data = RSTData()
    model = AttnSegModel(args)
    model.restore('best', args.model_dir)
    if model.use_ema:
        model.sess.run(model.ema_backup_op)
        model.sess.run(model.ema_assign_op)
    return rst_data, model


def segment_batches(data_batches):
    edus = []
    for batch in data_batches:
        batch_pred_segs = model.segment(batch)
        for sample, pred_segs in zip(batch['raw_data'], batch_pred_segs):
            logging.info(f"{pred_segs}")
            one_edu_words = []
            for word_idx, word in enumerate(sample['words']):
                if word_idx in pred_segs:
                    edus.append(' '.join(one_edu_words))
                    one_edu_words = []
                one_edu_words.append(word)
            if one_edu_words:
                edus.append(' '.join(one_edu_words))
    return edus


def segment_text(rst_data):
    data_batches = rst_data.gen_mini_batches(batch_size=32, test=True, shuffle=False)
    return segment_batches(data_batches)


@app.on_event("startup")
async def startup_event():
    global model, rst_data, spacy_nlp
    args, _ = get_argparser().parse_known_args()
    rst_data, model = load_models(args)
    spacy_nlp = spacy.load('en', disable=['ner', 'textcat'])


@app.get("/", response_class=HTMLResponse)
def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class SegmentationRequest(BaseModel):
    text: str


@app.post("/api/segment")
async def get_segmentation(r: SegmentationRequest):
    """Description
    """
    sents = []
    for sent in spacy_nlp(r.text).sents:
        sents.append({
            'raw_text': sent.text,
            'words': [token.text for token in sent],
            'edu_seg_indices': []
        })
    rst_data.test_samples = sents
    data_batches = rst_data.gen_mini_batches(batch_size=32, test=True, shuffle=False)
    return {"segments": segment_batches(data_batches)}


@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument("hostname")
@click.option("--port", default=8080, type=int)
@click.option("--debug", is_flag=True)
@click.argument('timeit_args', nargs=-1, type=click.UNPROCESSED)
def main(hostname, port, debug, timeit_args):
    uvicorn.run("api:app", host=hostname, port=port, log_level="debug" if debug else "info", reload=debug)


if __name__ == '__main__':
    main()
