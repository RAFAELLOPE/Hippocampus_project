const { Router } = require('express');
const {query_model, upload} = require('../controllers/model_controllers')
const expressfileUpload = require('express-fileupload');

const router = Router();

// Pasamos por el middleware del express-fileupload
router.use(expressfileUpload()); // con esto atrapamos el fichero que manden

router.post('/', query_model);


// Postman Test
// Endpoint: http://localhost:3000/api/ai-model/upload
// Body: choose form-data. key: imagen, value: choose nifti file

router.post('/upload', upload);


module.exports = router;