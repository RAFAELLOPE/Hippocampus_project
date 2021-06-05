const { Router } = require('express');
const {query_model} = require('../controllers/model_controllers')

const router = Router();

router.post('/', query_model);

module.exports = router;