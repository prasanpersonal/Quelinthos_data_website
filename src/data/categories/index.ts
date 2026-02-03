/**
 * CATEGORY REGISTRY
 *
 * To add a new category:
 * 1. Copy _template.ts → XX-your-category.ts
 * 2. Fill in all fields (run `npm test` to validate)
 * 3. Import it below
 * 4. Add it to the allCategories array
 * 5. Run `npm test` to verify everything passes
 */

import { marketingCategory } from './01-marketing.ts';
import { salesCategory } from './02-sales.ts';
import { customerServiceCategory } from './03-customer-service.ts';
import { ecommerceCategory } from './04-ecommerce.ts';
import { supplyChainCategory } from './05-supply-chain.ts';
import { financeCategory } from './06-finance.ts';
import { productManagementCategory } from './07-product-management.ts';
import { legalCategory } from './08-legal.ts';
import { constructionCategory } from './09-construction.ts';
import { hrCategory } from './10-hr.ts';
import type { Category, PainPoint } from '../types.ts';

export const allCategories: Category[] = [
  marketingCategory,
  salesCategory,
  customerServiceCategory,
  ecommerceCategory,
  supplyChainCategory,
  financeCategory,
  productManagementCategory,
  legalCategory,
  constructionCategory,
  hrCategory,
];

export const allPainPoints: PainPoint[] = allCategories.flatMap(c => c.painPoints);

export function getCategoryBySlug(slug: string): Category | undefined {
  return allCategories.find(c => c.id === slug);
}

export function getPainPointBySlug(slug: string): PainPoint | undefined {
  return allPainPoints.find(pp => pp.id === slug);
}

export function getCategoryForPainPoint(painPointId: string): Category | undefined {
  return allCategories.find(c => c.painPoints.some(pp => pp.id === painPointId));
}
