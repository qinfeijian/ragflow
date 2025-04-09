import uuid
import time
import logging
from flask import request, jsonify

from api.db import LLMType
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle, TenantLLMService
from api import settings
from api.utils.api_utils import validate_request, build_error_result, apikey_required
from rag.app.tag import label_question


@manager.route('/kg/retrieval', methods=['POST'])  # noqa: F821
@apikey_required
@validate_request("knowledge_id", "query")
def retrieval(tenant_id):

    # 生成随机任务 ID
    task_id = uuid.uuid4().hex 
    logging.info(f"Task  ID: {task_id} - 请求参数：{request.json} - Request received")

    req = request.json
    question = req["query"]
    # kb_id = req["knowledge_id"]
    kb_ids = req.get("knowledge_id", [])
    logging.info(f"Task  ID: {task_id} - request: {question} - Request received")
   
    try:
        start_time = time.time()
        kbs = KnowledgebaseService.get_by_ids(kb_ids)
        logging.info(f"Task  ID: {task_id} - request: {question} - 获取文档信息耗时 {time.time()  - start_time:.4f} seconds")
        
        embd_nms = list(set([kb.embd_id for kb in kbs]))
        if len(embd_nms) != 1:
            return build_error_result(
                data=False, message='Knowledge bases use different embedding models or does not exist."',
                code=settings.RetCode.AUTHENTICATION_ERROR)

        embd_mdl = TenantLLMService.model_instance(
            kbs[0].tenant_id, LLMType.EMBEDDING.value, llm_name=kbs[0].embd_id)
        start_time = time.time()
        # if not e:
        #     return build_error_result(message="Knowledgebase not found!", code=settings.RetCode.NOT_FOUND)

        # if kb.tenant_id != tenant_id:
        #     return build_error_result(message="Knowledgebase not found!", code=settings.RetCode.NOT_FOUND)
        # start_time = time.time() 
        # embd_mdl = LLMBundle(kb.tenant_id, LLMType.EMBEDDING.value, llm_name=kb.embd_id)
        logging.info(f"Task  ID: {task_id} - request: {question} - Initializing embedding model took {time.time()  - start_time:.4f} seconds")
        start_time = time.time() 
        ck = settings.kg_retrievaler.retrieval(question,
                                                [tenant_id],
                                                kb_ids,
                                                embd_mdl,
                                                LLMBundle(kbs[0].tenant_id, LLMType.CHAT))
        logging.info(f"Task  ID: {task_id} - request: {question}- Retrieving knowledge graph took {time.time()  - start_time:.4f} seconds")
        records = []
        if ck["content_with_weight"]:
            records.append({
                "content": ck["content_with_weight"],
                "score": ck["similarity"],
                "title": ck["docnm_kwd"],
                "metadata": {}
            })
        return jsonify({"records": records})
    except Exception as e:
        if str(e).find("not_found") > 0:
            return build_error_result(
                message='No chunk found! Check the chunk status please!',
                code=settings.RetCode.NOT_FOUND
            )
        return build_error_result(message=str(e), code=settings.RetCode.SERVER_ERROR)
