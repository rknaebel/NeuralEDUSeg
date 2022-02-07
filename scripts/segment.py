import json
import logging
import sys

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from neuraleduseg.config import parse_args
from neuraleduseg.model.atten_seg import AttnSegModel
from neuraleduseg.rst_edu_reader import RSTData


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
        batch_edu_breaks = model.segment(batch)
        for sent, pred_segs in zip(batch['raw_data'], batch_edu_breaks):
            edu_sent = []
            edu_start = 0
            for edu_break in pred_segs:
                edu_sent.append({'start': edu_start, 'end': edu_break})
                edu_start = edu_break
            if edu_start != len(sent['words']):
                edu_sent.append({'start': edu_start, 'end': len(sent['words'])})
            edu_sents.append({'edu': edu_sent})
    return edu_sents


def segment_text(rst_data, model):
    data_batches = rst_data.gen_mini_batches(batch_size=32, test=True, shuffle=False)
    return segment_batches(model, data_batches)


def convert_input(doc):
    return [{
        'words': [t['surface'] for t in sent['tokens']],
        'edu_seg_indices': [],
    } for sent in doc['sentences']]


def main():
    # HACK STDOUT
    stdout = sys.stdout
    sys.stdout = sys.stderr
    # HACK END
    args = parse_args()
    rst_data, model = load_models(args)
    with open(args.input_file, 'r') as fh:
        for line in fh:
            doc = json.loads(line)
            rst_data.test_samples = convert_input(doc)
            result = segment_text(rst_data, model)
            for sent, edus in zip(doc['sentences'], result):
                sent['edu'] = edus['edu']
            stdout.write(json.dumps(doc) + '\n')


if __name__ == "__main__":
    main()
