# freight-price-prediction

Freight price regression model tracked in MLflow and served with FastAPI.

**Status: planned.** This repo is a work in progress; the checklist lives in the epic issue and on the [Portfolio build-out](https://github.com/users/gkhnelbstn/projects/6) board.

## Plan

- What: regression model for freight prices on synthetic or open data, tracked in MLflow and served with FastAPI.
- Must have: baseline vs gradient boosting, feature importance, model registry, `/predict` endpoint with input validation, Docker Compose, tests.
- Done when: `curl` to the endpoint returns a price and the README shows the error metrics honestly.
- Alternative: finish the prediction part of `end-to-end-video-game-ml` instead, since that repo's description still says "will use".
