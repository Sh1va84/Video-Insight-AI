import express from "express";
import { storeDocument } from "../services/storeDocumentService.js";

const router = express.Router();

// Handle store document route
router.post('/', async (req, res) => {
    try {
        const result = await storeDocument(req);
        res.status(200).json(result);
    } catch (error) {
        console.error("Error in storeDocument: ", error);
        res.status(500).json({
            error: "An error occurred during the request."
        })
    }
});

export default router;