
.PHONY: build test clean

JSONFILES=test/expectation.json test/initialize.json test/gaussian.json \
test/maximization.json test/inference.json test/fit.json \
test/sample.json

build: ${JSONFILES}

test: | build
	npm test

clean:
	rm -f ${JSONFILES}

test/sample.json: test/sample.py test/tool_generate_data.py
	python3 -m test.sample

test/fit.json: test/fit.py test/tool_generate_data.py
	python3 -m test.fit

test/inference.json: test/inference.py test/tool_generate_data.py
	python3 -m test.inference

test/maximization.json: test/maximization.py test/tool_generate_data.py
	python3 -m test.maximization

test/expectation.json: test/expectation.py test/tool_generate_data.py
	python3 -m test.expectation

test/initialize.json: test/initialize.py test/tool_generate_data.py
	python3 -m test.initialize

test/gaussian.json: test/gaussian.py test/tool_generate_data.py
	python3 -m test.gaussian
