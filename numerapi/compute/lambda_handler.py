import pandas as pd
from numerapi import NumerAPI
import boto3
import json
import logging
import sys
import traceback

logger = logging.getLogger()
logger.setLevel(logging.INFO)

secretsmanager = boto3.client('secretsmanager')
api_keys_secret = secretsmanager.get_secret_value(SecretId='numerai-api-keys')
secret = json.loads(api_keys_secret['SecretString'])


def run(event, context):

    napi = NumerAPI(
        public_id=secret['public_id'],
        secret_key=secret['secret_key']
    )

    model_id = event['model_id']
    if 'data_version' in event:
        data_version = event['data_version']
    else:
        data_version = 'v4'

    print(f'Running submission for model_id: {model_id}')
    print(f'Data version: {data_version}')

    request_id = context.aws_request_id
    log_stream_name = context.log_stream_name
    set_lambda_status(context.function_name, model_id, request_id, "in_progress", napi, log_stream_name)

    try:

        current_round = napi.get_current_round()

        if data_version in ['v2', 'v3']:
            live_data_filename = 'numerai_live_data.parquet'
        else:
            live_data_filename = f'live.parquet'

        live_data_local_path = f"/tmp/{data_version}/{live_data_filename}"
        napi.download_dataset(f"{data_version}/{live_data_filename}", live_data_local_path)
        live_data = pd.read_parquet(live_data_local_path)
        logger.info(f'Downloaded {live_data_local_path}')

        s3 = boto3.client('s3')
        aws_account_id = boto3.client('sts').get_caller_identity().get('Account')
        s3.download_file(f'numerai-compute-{aws_account_id}', f'{model_id}/model.pkl', '/tmp/model.pkl')
        model = pd.read_pickle(f"/tmp/model.pkl")
        logger.info(f'Unpickled {model_id}/model.pkl')

        model_name = 'model'
        s3.download_file(f'numerai-compute-{aws_account_id}', f'{model_id}/features.json', '/tmp/features.json')
        f = open('/tmp/features.json')
        features = json.load(f)
        logger.info(f'Loaded features {model_id}/features.json')

        live_data.loc[:, f"preds_{model_name}"] = model.predict(
            live_data.loc[:, features])

        live_data["prediction"] = live_data[f"preds_{model_name}"].rank(pct=True)
        logger.info(f'Live predictions and ranked')

        predict_output_path = f"/tmp/live_predictions_{current_round}.csv"
        if data_version == 'v2':
            # v2 live data id column is not the index so needs to be specified in output here
            live_data[["id", "prediction"]].to_csv(predict_output_path, index=False)
        else:
            live_data["prediction"].to_csv(predict_output_path)

        print(f'submitting {predict_output_path}')
        napi.upload_predictions(predict_output_path, model_id=model_id)
        print('submission complete!')

    except Exception as ex:
        set_lambda_status(context.function_name, model_id, request_id, "error", napi, log_stream_name)
        exception_type, exception_value, exception_traceback = sys.exc_info()
        traceback_string = traceback.format_exception(exception_type, exception_value, exception_traceback)
        err_msg = json.dumps({
            "errorType": exception_type.__name__,
            "errorMessage": str(exception_value),
            "stackTrace": traceback_string
        })
        logger.error(err_msg)
        return False

    set_lambda_status(context.function_name, model_id, request_id, "complete", napi, log_stream_name)

    return True


def run_diagnostics(model_id, napi):
    # download validation data
    # predict on validation data
    # diagnostics_id = napi.upload_diagnostics(file_path, tournament=8, model_id=model_id)
    pass


def set_lambda_status(function_name, model_id, request_id, status, napi, log_stream_name=None):
    query = f'''
        mutation setModelLambdaStatus($function_name: String!, $model_id: String!, $request_id: String!, $status: String!, $log_stream_name: String) {{
          modelLambdaStatus(
            functionName: $function_name, 
            modelId: $model_id, 
            requestId: $request_id, 
            status: $status,
            logStreamName: $log_stream_name) {{
            requestId
          }}
        }}
        '''
    napi.raw_query(
        query=query,
        authorization=True,
        variables={
            'function_name': function_name,
            'model_id': model_id,
            'request_id': request_id,
            'status': status,
            'log_stream_name': log_stream_name
        }
    )


if __name__ == '__main__':
    run({}, {})
