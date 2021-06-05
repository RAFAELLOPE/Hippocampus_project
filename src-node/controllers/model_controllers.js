
const query_model = (req, res) => {
    const { start } = req.query;
    //Send message to rabbit mq so as for the python process to start infering the images
    res.json({
        status:'OK',
        start
    });
}

module.exports = {query_model};