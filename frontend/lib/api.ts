const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export interface PredictionInput {
    sample_id?: number;
    features?: number[];
}

export interface PredictionOutput {
    estimated_glucose: number;
    glucose_range: string;
    confidence: number;
    model_used: string;
    notes: string;
}

export async function getPrediction(input: PredictionInput): Promise<PredictionOutput> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(input),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
    }

    return response.json();
}

export async function getSystemInfo() {
    const response = await fetch(`${API_BASE_URL}/info`);
    return response.json();
}
