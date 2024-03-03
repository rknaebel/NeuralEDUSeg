import glob
import logging
import os
import pathlib
import re
import sys
import zipfile

import spacy
from conllu import TokenList
from conllu.models import Token, Metadata
from conllu.parser import DEFAULT_FIELDS

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf

from neuraleduseg.config import parse_args
from neuraleduseg.model.atten_seg import AttnSegModel
from neuraleduseg.rst_edu_reader import RSTData

RE_PAR = re.compile(r"\n\s*(?:\n|\s\s\s|\t)\s*")
RE_CR = re.compile(r"\r+")
RE_WS_PLUS = re.compile(r"\s+")


def split_to_paragraphs(content):
    res = RE_CR.sub(' ', content)
    res = RE_PAR.split(res)
    res = filter(lambda x: x != '.START', map(lambda x: RE_WS_PLUS.sub(' ', x).strip(), res))
    return list(res)


def load_models(args):
    """Load models needed for EDU segmentation."""
    rst_data = RSTData()
    model = AttnSegModel(args)
    model.restore('best', args.model_dir)
    if model.use_ema:
        model.sess.run(model.ema_backup_op)
        model.sess.run(model.ema_assign_op)
    return rst_data, model


def segment_batches(model, data_batches):
    edu_sents = []
    for batch in data_batches:
        try:
            batch_edu_breaks = model.segment(batch)
            for sent, pred_segs in zip(batch['raw_data'], batch_edu_breaks):
                edu_sent = []
                edu_start = 0
                len_sent = sum(t != '' for t in sent['words'])
                for edu_break in pred_segs:
                    edu_sent.append({'start': edu_start, 'end': edu_break})
                    edu_start = edu_break
                if edu_start != len_sent:
                    edu_sent.append({'start': edu_start, 'end': len_sent})
                edu_sents.append(edu_sent)
        except:
            for sent in batch['raw_data']:
                sys.stderr.write(f">> {sent}")
                edu_sents.append([])
    return edu_sents


def segment_text(rst_data, model):
    data_batches = rst_data.gen_mini_batches(batch_size=32, test=True, shuffle=False)
    return segment_batches(model, data_batches)


def convert_input(doc):
    return [{
        'words': [t['surface'] for t in sent['tokens']],
        'edu_seg_indices': [],
    } for sent in doc['sentences']]


def extract(source_path: str):
    with zipfile.ZipFile(source_path) as zh:
        for fn in zh.filelist:
            if not fn.filename.endswith('.txt'):
                continue
            text = zh.open(fn).read().decode('latin-1')
            title, text = text.split('\n\n', maxsplit=1)
            corpus, topic, id = fn.filename.split('/')[-3:]
            yield {
                'meta': {
                    'title': title.strip(),
                    'topic': topic,
                    'corpus': corpus,
                },
                'text': text.strip(),
            }


def main():
    logging.basicConfig(level=logging.INFO)
    tf.compat.v1.logging.set_verbosity('ERROR')

    # HACK STDOUT
    stdout = sys.stdout
    sys.stdout = sys.stderr
    # HACK END

    args = parse_args()
    logger = logging.getLogger("SegEDU")
    logger.info('Running with args : {}'.format(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    rst_data, model = load_models(args)
    spacy_nlp = spacy.load('en', disable=['ner', 'textcat'])

    for input_file in glob.glob(args.input_file):
        fin = pathlib.Path(input_file)
        fout = pathlib.Path(args.result_dir) / fin.with_suffix('.conll').name
        sents = []
        paragraphs = split_to_paragraphs(fin.open(encoding='latin-1').read())
        paragraph_counts = []
        for content in paragraphs:
            doc_sents = list(spacy_nlp(content).sents)
            for sent in doc_sents:
                sents.append({
                    'raw_text': sent.text,
                    'words': [token.text for token in sent],
                    'edu_seg_indices': []
                })
            paragraph_counts.append(len(doc_sents))
        rst_data.test_samples = sents
        data_batches = rst_data.gen_mini_batches(batch_size=32, test=True, shuffle=False)
        segments = segment_batches(model, data_batches)
        result = []
        doc_id = fin.name
        par_i = 1
        meta = {
            'newdoc id': doc_id,
            'newpar id': f'{doc_id}-p{par_i}'
        }
        par_ctr = 0
        par_cur = paragraph_counts.pop()
        for sent_i, (sent, edus) in enumerate(zip(sents, segments)):
            tokens = []
            for edu in edus:
                for tok_i in range(edu['start'], edu['end']):
                    misc = {}
                    if tok_i == edu['start']:
                        misc['BeginSeg'] = "YES"
                    if par_ctr >= par_cur:
                        meta['newpar id'] = f'{doc_id}-p{par_i}'
                        par_i += 1
                        par_ctr = 0
                        par_cur = paragraph_counts.pop()
                    tokens.append(Token(
                        id=tok_i + 1,
                        form=sent['words'][tok_i],
                        lemma="_",
                        upos="_",
                        xpos="_",
                        feats='_',
                        head="_",
                        deprel="_",
                        deps='_',
                        misc=misc))
            meta['sent_id'] = str(sent_i)
            meta['text'] = sent['raw_text']
            result.append(TokenList(tokens, metadata=Metadata(meta), default_fields=DEFAULT_FIELDS))
            meta = {}
            par_ctr += 1
        with fout.open('w') as fh:
            fh.write(''.join(sent.serialize() for sent in result))


if __name__ == "__main__":
    main()
