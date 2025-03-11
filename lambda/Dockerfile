FROM public.ecr.aws/lambda/python:3.9

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies into /var/lang/lib/python3.9/site-packages
RUN pip install -r requirements.txt

# Copy your function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.handler" ]
