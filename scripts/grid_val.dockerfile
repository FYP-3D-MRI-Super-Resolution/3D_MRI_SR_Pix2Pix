FROM pix_base

WORKDIR Pix2PixNIfTI
ADD . .

RUN chmod +x ./scripts/grid_search_test.sh

ENTRYPOINT ["bash", "./scripts/grid_search_test.sh"]