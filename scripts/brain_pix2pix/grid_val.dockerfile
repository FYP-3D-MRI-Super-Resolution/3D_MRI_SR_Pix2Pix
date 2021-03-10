FROM pix_base

WORKDIR Pix2PixNIfTI
ADD . .

RUN chmod +x ./scripts/brain_pix2pix/grid_search_test.sh

ENTRYPOINT ["bash", "/Pix2PixNIfTI/scripts/brain_pix2pix/grid_search_test.sh"]