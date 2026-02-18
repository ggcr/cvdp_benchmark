[NotSoTiny](https://huggingface.co/datasets/HPAI-BSC/NotSoTiny-25-12) is a large, living benchmark for RTL code generation made by [HPAI-BSC](https://huggingface.co/HPAI-BSC). It's built from 1,114 real hardware designs produced by the Tiny Tapeout community. This integration into the [CVDP framework](https://github.com/NVlabs/cvdp_benchmark) was developed as part of the [Si2 Extend and Expand](https://si2.org/) work group.

## Subsets and Shuttles

The default dataset contains all [NotSoTiny-25-12](https://huggingface.co/datasets/HPAI-BSC/NotSoTiny-25-12) Tiny Tapeout shuttles combined onto a single dataset comprising 1114 total tasks. However, you can also access individual shuttles as subsets.

| Subset | # Tasks | Launched date | Tiny Tapeout source |
|----|---|---|---|
| `default`  | 1,114 | - | - |
| `tt06` | 108 | 2024-01-30 | https://tinytapeout.com/chips/tt06/ |
| `tt07` | 177 | 2024-04-22 | https://tinytapeout.com/chips/tt07/ | 
| `tt08` | 196 | 2024-06-10 | https://tinytapeout.com/chips/tt08/  |
| `tt09` | 250 | 2024-09-07 | https://tinytapeout.com/chips/tt09/ |
| `tt10` | 214 | 2025-03-12 | https://tinytapeout.com/chips/ttihp25a/ |
| `ttsky` | 169 | 2025-06-27 | https://tinytapeout.com/chips/ttsky25a/ |

## Run it with CVDP

**1.** Clone the CVDP framework:

	$ git clone https://github.com/NVlabs/cvdp_benchmark.git
	$ cd cvdp_benchmark/
	$ uv init && uv add -r requirements.txt

**2.** Download the dataset shuttles:

	$ git xet install
	$ git clone https://huggingface.co/datasets/HPAI-BSC/NotSoTiny-25-12-CVDP
	$ ls NotSoTiny-25-12-CVDP/shuttles/
		tt06.jsonl   tt07.jsonl   tt08.jsonl   tt09.jsonl   tt10.jsonl   ttsky.jsonl
    $ cp .env.example .env && echo "OSS_SIM_IMAGE=ggcr0/turtle-eval:2.3.4" >> .env

**3.** [Optional] Validate with golden solutions:

	$ uv run run_benchmark.py -f NotSoTiny-25-12-CVDP/shuttles/tt06.jsonl

**4.** Run the benchmark (inference + eval):

    $ export OPENROUTER_API_KEY=sk-or-v1-... 
	$ uv run run_samples.py \
  	     -f NotSoTiny-25-12-CVDP/shuttles/tt06.jsonl \
  	     -l \
  	     -m mistralai/codestral-2508 \
  	     -c examples/openrouter_factory.py \
  	     -n 3 \
  	     -k 1

You can repeat this process for any other shuttle present in the `shuttles/` dir.

We also recommend the usage of the `-t <workers>` flag to speed-up the process.

