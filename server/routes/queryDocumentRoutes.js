import express from "express";
import { queryDocument } from "../services/queryDocumentService.js";

const router = express.Router();

// Handle store document route
router.post('/', async (req, res) => {
    try {
        const result = await queryDocument(req);
        res.setHeader('Content-Type', 'text/event-stream')
        res.setHeader('Cache-Control', 'no-cache')
        res.setHeader('Connection', 'keep-alive')
        // result.pipe(res);
        res.status(200).json(result);
    } catch (error) {
        console.error("Error in queryDocument: ", error);
        res.status(500).json({
            error: "An error occurred during the request."
        })
    }
});

export default router;