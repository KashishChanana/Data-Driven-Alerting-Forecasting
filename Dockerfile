FROM continuumio/anaconda3

LABEL maintainer = "Kashish Chanana <kchanana@ebay.com>" \
      description = "Machine Learning Alert Enhancement Intern Project, Summer 2021"

WORKDIR .

COPY . .

EXPOSE 1234

RUN conda env update -n base --file environ.yml

RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/base/bin:$PATH

RUN echo "Activated"

CMD python3 main.py --connect 'y' --feecode 0000 --siteid 0 --multi 'y' --WoW y --hourly 12 --model_name Prophet-Multi --substier y


