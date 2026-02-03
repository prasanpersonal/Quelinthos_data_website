/**
 * SME PAIN POINT REGISTRY
 *
 * To add a new SME pain point:
 * 1. Copy _template.ts → XX-your-pain-point.ts
 * 2. Fill in all fields (run `npm test` to validate)
 * 3. Import it below
 * 4. Add it to the named exports and to the allSmePainPoints array
 * 5. Run `npm test` to verify everything passes
 */

import type { SmePainPoint } from '../types.ts';

import { invoiceProcessing } from './01-invoice-processing.ts';
import { reportGeneration } from './02-report-generation.ts';
import { duplicateRecords } from './03-duplicate-records.ts';
import { dataMigration } from './04-data-migration.ts';
import { spreadsheetDependency } from './05-spreadsheet-dependency.ts';
import { apiIntegration } from './06-api-integration.ts';
import { customerChurn } from './07-customer-churn.ts';
import { inventoryForecasting } from './08-inventory-forecasting.ts';
import { complianceReporting } from './09-compliance-reporting.ts';
import { hrTurnover } from './10-hr-turnover.ts';

export {
  invoiceProcessing,
  reportGeneration,
  duplicateRecords,
  dataMigration,
  spreadsheetDependency,
  apiIntegration,
  customerChurn,
  inventoryForecasting,
  complianceReporting,
  hrTurnover,
};

export const allSmePainPoints: SmePainPoint[] = [
  invoiceProcessing,
  reportGeneration,
  duplicateRecords,
  dataMigration,
  spreadsheetDependency,
  apiIntegration,
  customerChurn,
  inventoryForecasting,
  complianceReporting,
  hrTurnover,
];

export function getSmePainPointById(id: string): SmePainPoint | undefined {
  return allSmePainPoints.find((painPoint) => painPoint.id === id);
}
