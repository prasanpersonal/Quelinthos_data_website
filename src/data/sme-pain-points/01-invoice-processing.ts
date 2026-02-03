import type { SmePainPoint } from '../types.ts';

export const invoiceProcessing: SmePainPoint = {
  id: 'invoice-processing',
  number: 1,
  title: 'Invoice Processing Bottleneck',
  problem:
    'Manual invoice data entry takes 15 minutes per invoice. With 500+ invoices monthly, your AP team spends 60% of their time on data entry.',
  solution:
    'OCR + AI extraction pipeline that reads invoices, matches POs, and posts to your accounting system automatically.',
  impact:
    'Reduce processing time by 85%. Save 120+ hours monthly. Eliminate 95% of data entry errors.',
  icon: 'FileText',
};
