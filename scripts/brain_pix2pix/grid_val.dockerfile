FROM pix_base

RUN chmod +x ./scripts/brain_pix2pix/grid_search_test.sh

ENTRYPOINT ["bash", "/Pix2PixNIfTI/scripts/brain_pix2pix/grid_search_test.sh"]