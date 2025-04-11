# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: face_recognition.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16\x66\x61\x63\x65_recognition.proto\x12\x0f\x66\x61\x63\x65_recognition\"\xaf\x01\n\x04\x46\x61\x63\x65\x12\x0f\n\x07\x66\x61\x63\x65_id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x11\n\tembedding\x18\x03 \x03(\x02\x12*\n\x04rect\x18\x04 \x01(\x0b\x32\x1c.face_recognition.FaceRect\x12\x32\n\x08metadata\x18\x05 \x01(\x0b\x32 .face_recognition.FaceMetadata\x12\x15\n\rquality_score\x18\x06 \x01(\x02\"?\n\x08\x46\x61\x63\x65Rect\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\"\xef\x01\n\x0c\x46\x61\x63\x65Metadata\x12\x15\n\rregistered_by\x18\x01 \x01(\t\x12\x11\n\tdevice_id\x18\x02 \x01(\t\x12\x19\n\x11registration_date\x18\x03 \x01(\t\x12\x15\n\rlast_accessed\x18\x04 \x01(\t\x12G\n\x0f\x61\x64\x64itional_info\x18\x05 \x03(\x0b\x32..face_recognition.FaceMetadata.AdditionalInfoEntry\x1a:\n\x13\x41\x64\x64itionalInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x07\x38\x01\"\x91\x01\n\x12RegisterFaceRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x32\n\x08metadata\x18\x03 \x01(\x0b\x32 .face_recognition.FaceMetadata\x12\x17\n\x0fverify_liveness\x18\x04 \x01(\x08\"j\n\x13RegisterFaceResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07\x66\x61\x63\x65_id\x18\x02 \x01(\t\x12\x0f\n\x07message\x18\x03 \x01(\t\x12\x15\n\rquality_score\x18\x04 \x01(\x02\"T\n\x10RecognizeRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x11\n\tthreshold\x18\x02 \x01(\x02\x12\x13\n\x0bmax_results\x18\x03 \x01(\x05\"F\n\x11RecognizeResponse\x12\x31\n\x07matches\x18\x01 \x03(\x0b\x32 .face_recognition.FaceMatch\"Y\n\tFaceMatch\x12%\n\x04\x66\x61\x63\x65\x18\x01 \x01(\x0b\x32\x17.face_recognition.Face\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\x11\n\tdistance\x18\x03 \x01(\x02\"4\n\x10ListFacesRequest\x12\x0e\n\x06offset\x18\x01 \x01(\x05\x12\r\n\x05limit\x18\x02 \x01(\x05\"P\n\x11ListFacesResponse\x12(\n\x05\x66\x61\x63\x65s\x18\x01 \x03(\x0b\x32\x19.face_recognition.Face\x12\r\n\x05total\x18\x02 \x01(\x05\"\"\n\x10\x44\x65leteFaceRequest\x12\x0e\n\x06\x66\x61\x63\x65_id\x18\x01 \x01(\t\"9\n\x11\x44\x65leteFaceResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"c\n\x14UpdateMetadataRequest\x12\x0f\n\x07\x66\x61\x63\x65_id\x18\x01 \x01(\t\x12:\n\x08metadata\x18\x02 \x01(\x0b\x32(.face_recognition.FaceMetadata\"=\n\x15UpdateMetadataResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"$\n\x11\x46\x61\x63\x65QualityRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\"\xc3\x01\n\x12\x46\x61\x63\x65QualityResponse\x12\x15\n\rquality_score\x18\x01 \x01(\x02\x12K\n\x0equality_factors\x18\x02 \x03(\x0b\x32\x33.face_recognition.FaceQualityResponse.QualityFactorsEntry\x1aI\n\x13QualityFactorsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12!\n\x05value\x18\x02 \x01(\x0b\x32\x12.face_recognition.QualityFactor:\x02\x38\x01\"0\n\rQualityFactor\x12\r\n\x05value\x18\x01 \x01(\x02\x12\x10\n\x08\x63\x61tegory\x18\x02 \x01(\t\"9\n\x0eLivenessRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x12\n\nsession_id\x18\x02 \x01(\t\"Q\n\x0fLivenessResponse\x12\x0f\n\x07is_live\x18\x01 \x01(\x08\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\x0f\n\x07message\x18\x03 \x01(\t2\x9d\x05\n\x16\x46\x61\x63\x65RecognitionService\x12\\\n\x0cRegisterFace\x12#.face_recognition.RegisterFaceRequest\x1a$.face_recognition.RegisterFaceResponse\"\x01\x12V\n\x0eRecognizeFaces\x12!.face_recognition.RecognizeRequest\x1a\".face_recognition.RecognizeResponse\x12S\n\tListFaces\x12!.face_recognition.ListFacesRequest\x1a\".face_recognition.ListFacesResponse"\x01\x12V\n\nDeleteFace\x12!.face_recognition.DeleteFaceRequest\x1a\".face_recognition.DeleteFaceResponse\"\x01\x12\x65\n\x11UpdateFaceMetadata\x12%.face_recognition.UpdateMetadataRequest\x1a&.face_recognition.UpdateMetadataResponse\"\x01\x12Y\n\x0eGetFaceQuality\x12\".face_recognition.FaceQualityRequest\x1a#.face_recognition.FaceQualityResponse\x12P\n\x0eVerifyLiveness\x12\x1f.face_recognition.LivenessRequest\x1a\x1d.face_recognition.LivenessResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'face_recognition_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _FACEMETADATA_ADDITIONALINFOENTRY._options = None
  _FACEMETADATA_ADDITIONALINFOENTRY._serialized_options = b'8\001'
  _FACEQUALITYRESPONSE_QUALITYFACTORSENTRY._options = None
  _FACEQUALITYRESPONSE_QUALITYFACTORSENTRY._serialized_options = b'8\001'
  _globals['_FACE']._serialized_start=44
  _globals['_FACE']._serialized_end=219
  _globals['_FACERECT']._serialized_start=221
  _globals['_FACERECT']._serialized_end=284
  _globals['_FACEMETADATA']._serialized_start=287
  _globals['_FACEMETADATA']._serialized_end=526
  _globals['_FACEMETADATA_ADDITIONALINFOENTRY']._serialized_start=468
  _globals['_FACEMETADATA_ADDITIONALINFOENTRY']._serialized_end=526
  _globals['_REGISTERFACEREQUEST']._serialized_start=529
  _globals['_REGISTERFACEREQUEST']._serialized_end=674
  _globals['_REGISTERFACERESPONSE']._serialized_start=676
  _globals['_REGISTERFACERESPONSE']._serialized_end=782
  _globals['_RECOGNIZEREQUEST']._serialized_start=784
  _globals['_RECOGNIZEREQUEST']._serialized_end=868
  _globals['_RECOGNIZERESPONSE']._serialized_start=870
  _globals['_RECOGNIZERESPONSE']._serialized_end=940
  _globals['_FACEMATCH']._serialized_start=942
  _globals['_FACEMATCH']._serialized_end=1031
  _globals['_LISTFACESREQUEST']._serialized_start=1033
  _globals['_LISTFACESREQUEST']._serialized_end=1085
  _globals['_LISTFACESRESPONSE']._serialized_start=1087
  _globals['_LISTFACESRESPONSE']._serialized_end=1167
  _globals['_DELETEFACEREQUEST']._serialized_start=1169
  _globals['_DELETEFACEREQUEST']._serialized_end=1203
  _globals['_DELETEFACERESPONSE']._serialized_start=1205
  _globals['_DELETEFACERESPONSE']._serialized_end=1262
  _globals['_UPDATEMETADATAREQUEST']._serialized_start=1264
  _globals['_UPDATEMETADATAREQUEST']._serialized_end=1363
  _globals['_UPDATEMETADATARESPONSE']._serialized_start=1365
  _globals['_UPDATEMETADATARESPONSE']._serialized_end=1426
  _globals['_FACEQUALITYREQUEST']._serialized_start=1428
  _globals['_FACEQUALITYREQUEST']._serialized_end=1464
  _globals['_FACEQUALITYRESPONSE']._serialized_start=1467
  _globals['_FACEQUALITYRESPONSE']._serialized_end=1662
  _globals['_FACEQUALITYRESPONSE_QUALITYFACTORSENTRY']._serialized_start=1589
  _globals['_FACEQUALITYRESPONSE_QUALITYFACTORSENTRY']._serialized_end=1662
  _globals['_QUALITYFACTOR']._serialized_start=1664
  _globals['_QUALITYFACTOR']._serialized_end=1712
  _globals['_LIVENESSREQUEST']._serialized_start=1714
  _globals['_LIVENESSREQUEST']._serialized_end=1771
  _globals['_LIVENESSRESPONSE']._serialized_start=1773
  _globals['_LIVENESSRESPONSE']._serialized_end=1854
  _globals['_FACERECOGNITIONSERVICE']._serialized_start=1857
  _globals['_FACERECOGNITIONSERVICE']._serialized_end=2526
