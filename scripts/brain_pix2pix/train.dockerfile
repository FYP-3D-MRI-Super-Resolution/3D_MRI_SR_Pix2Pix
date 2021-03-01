FROM pix_base

ADD ./scripts/brain_pix2pix ./scripts/brain_pix2pix/

RUN chmod +x ./scripts/brain_pix2pix/grid_search.sh

ENTRYPOINT ["bash", "/Pix2PixNIfTI/scripts/brain_pix2pix/grid_search.sh"]