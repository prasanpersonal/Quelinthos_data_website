import type { SmePainPoint } from '../types.ts';

export const inventoryForecasting: SmePainPoint = {
  id: 'inventory-forecasting',
  number: 8,
  title: 'Inventory Forecasting Guesswork',
  problem:
    'Your purchasing team orders based on gut feel and last year\'s numbers. Seasonal patterns, trends, and supplier lead times aren\'t factored in.',
  solution:
    'ML-powered demand forecasting that accounts for seasonality, trends, promotions, and supplier constraints.',
  impact:
    'Reduce stockouts by 40%. Cut excess inventory by 30%. Free up working capital trapped in overstock.',
  icon: 'Boxes',
};
