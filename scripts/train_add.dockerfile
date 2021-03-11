FROM pix_base

WORKDIR Pix2PixNIfTI
ADD . .

RUN chmod +x ./scripts/grid_search_add.sh

ENTRYPOINT ["bash", "/Pix2PixNIfTI/scripts/grid_search_add.sh"]